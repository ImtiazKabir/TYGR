#!/usr/bin/env python3
"""Diagnostic script to check DWARF parsing with current pyelftools version."""

import sys
from elftools.elf.elffile import ELFFile
from elftools.dwarf.locationlists import LocationParser
from elftools.dwarf.descriptions import ExprDumper, describe_form_class

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_dwarf.py <binary>")
        sys.exit(1)

    binary_path = sys.argv[1]
    print(f"Analyzing: {binary_path}")

    # Check pyelftools version
    import elftools
    print(f"pyelftools version: {elftools.__version__}")

    with open(binary_path, "rb") as f:
        elf = ELFFile(f)

        if not elf.has_dwarf_info():
            print("ERROR: No DWARF info in binary")
            sys.exit(1)

        dwarf_info = elf.get_dwarf_info()
        print(f"DWARF info found")

        # Count CUs
        cu_count = 0
        subprog_count = 0
        subprog_with_pc = 0

        for cu in dwarf_info.iter_CUs():
            cu_count += 1
            cu_die = cu.get_top_DIE()

            # Iterate through DIEs to find subprograms
            def iter_die(die):
                yield die
                for child in die.iter_children():
                    yield from iter_die(child)

            for die in iter_die(cu_die):
                if die.tag == "DW_TAG_subprogram":
                    subprog_count += 1
                    func_name = die.attributes.get("DW_AT_name")
                    if func_name:
                        func_name = func_name.value
                        if isinstance(func_name, bytes):
                            func_name = func_name.decode("utf-8")

                    # Try to get low/high PC
                    low_high_pc = None

                    # Method 1: DW_AT_ranges
                    if "DW_AT_ranges" in die.attributes:
                        try:
                            range_lists = dwarf_info.range_lists()
                            ranges_offset = die.attributes["DW_AT_ranges"].value
                            ranges = range_lists.get_range_list_at_offset(ranges_offset)

                            print(f"\n  Subprogram: {func_name}")
                            print(f"    Has DW_AT_ranges, offset={ranges_offset}")
                            print(f"    Range list type: {type(ranges)}")

                            if ranges:
                                first_range = ranges[0]
                                print(f"    First range entry type: {type(first_range)}")
                                print(f"    First range entry attributes: {dir(first_range)}")

                                # Check for different attribute names
                                if hasattr(first_range, 'begin_offset'):
                                    print(f"    Has begin_offset: {first_range.begin_offset}")
                                if hasattr(first_range, 'end_offset'):
                                    print(f"    Has end_offset: {first_range.end_offset}")
                                if hasattr(first_range, 'begin_address'):
                                    print(f"    Has begin_address: {first_range.begin_address}")
                                if hasattr(first_range, 'end_address'):
                                    print(f"    Has end_address: {first_range.end_address}")
                                if hasattr(first_range, 'start_offset'):
                                    print(f"    Has start_offset: {first_range.start_offset}")

                                subprog_with_pc += 1
                        except Exception as e:
                            print(f"\n  Subprogram: {func_name}")
                            print(f"    ERROR processing ranges: {e}")

                    # Method 2: DW_AT_low_pc / DW_AT_high_pc
                    elif "DW_AT_low_pc" in die.attributes:
                        low_pc = die.attributes["DW_AT_low_pc"].value

                        if "DW_AT_high_pc" in die.attributes:
                            highpc_attr = die.attributes["DW_AT_high_pc"]
                            highpc_class = describe_form_class(highpc_attr.form)

                            if highpc_class == "address":
                                high_pc = highpc_attr.value
                            elif highpc_class == "constant":
                                high_pc = low_pc + highpc_attr.value
                            else:
                                high_pc = None

                            if high_pc:
                                print(f"\n  Subprogram: {func_name}")
                                print(f"    low_pc={hex(low_pc)}, high_pc={hex(high_pc)}")
                                subprog_with_pc += 1
                    else:
                        print(f"\n  Subprogram: {func_name} - NO PC INFO")

        print(f"\n=== Summary ===")
        print(f"Compilation Units: {cu_count}")
        print(f"Total subprograms: {subprog_count}")
        print(f"Subprograms with PC range: {subprog_with_pc}")

        # Now test TYGR's dwarf parsing
        print(f"\n=== Testing TYGR's DWARF parsing ===")
        try:
            from src.analysis.dwarf import get_dwarf_info, dwarf_info_to_subprograms, dwarf_info_to_context, dwarf_subprogram_to_vars

            dwarf_info2 = get_dwarf_info(binary_path)
            dwarf_ctx = dwarf_info_to_context(dwarf_info2)

            subprog_list = list(dwarf_info_to_subprograms(dwarf_info2))
            print(f"TYGR found {len(subprog_list)} subprograms")

            for subprog in subprog_list:
                (directory, file_name, func_name, low_high_pc, cu_offset, subprog_die) = subprog
                print(f"\n  Function: {func_name}")
                print(f"    PC range: {hex(low_high_pc[0])} - {hex(low_high_pc[1])}")

                # Try to get variables
                try:
                    vars_list = list(dwarf_subprogram_to_vars(subprog, dwarf_ctx, dwarf_info2))
                    print(f"    Variables found: {len(vars_list)}")
                    for var in vars_list[:5]:  # Show first 5
                        var_name, var_locs, var_type, func_param = var
                        print(f"      - {var_name}: {var_type} (param={func_param})")
                    if len(vars_list) > 5:
                        print(f"      ... and {len(vars_list) - 5} more")
                except Exception as e:
                    print(f"    ERROR getting variables: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"ERROR in TYGR parsing: {e}")
            import traceback
            traceback.print_exc()

        # Test symbolic execution
        print(f"\n=== Testing Symbolic Execution ===")
        try:
            import angr
            print(f"angr version: {angr.__version__}")

            LOAD_OPTIONS = {
                "main_opts": {"base_addr": 0x0},
                "auto_load_libs": False
            }
            proj = angr.Project(binary_path, load_options=LOAD_OPTIONS)
            print(f"Project loaded: {proj}")
            print(f"Architecture: {proj.arch}")

            # Test CFG generation
            print(f"\nGenerating CFG...")
            cfg = proj.analyses.CFGFast()
            print(f"CFG generated")
            print(f"CFG functions: {len(cfg.functions)}")
            print(f"CFG model type: {type(cfg.model)}")

            # Check if main function is in CFG
            main_addr = 0x10908
            func = cfg.functions.get(main_addr)
            if func:
                print(f"\nMain function found in CFG at {hex(main_addr)}")
                print(f"  Function name: {func.name}")
                print(f"  Function size: {func.size}")
            else:
                print(f"\nWARNING: Main function NOT found at {hex(main_addr)}")
                print(f"Available functions:")
                for addr, fn in list(cfg.functions.items())[:10]:
                    print(f"  {hex(addr)}: {fn.name}")

            # Test the dominator strategy
            print(f"\n=== Testing LessSimpleDominatorStrategy ===")
            from src.analysis.angr.sim_exec import LessSimpleDominatorStrategy

            strategy = LessSimpleDominatorStrategy(proj, config=None)
            print(f"Strategy created")
            print(f"Strategy CFG functions: {len(strategy.cfg.functions)}")

            # Try sim_exec_function
            print(f"\nAttempting symbolic execution of main...")
            try:
                result = strategy.sim_exec_function(main_addr)
                print(f"Symbolic execution result: {result}")
                print(f"Number of state tuples: {len(result.tups)}")
            except Exception as e:
                print(f"ERROR in sim_exec_function: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"ERROR in symbolic execution test: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
